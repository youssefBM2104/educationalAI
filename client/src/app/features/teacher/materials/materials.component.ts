import {
  ChangeDetectionStrategy, Component, DestroyRef,
  OnInit, computed, inject, signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval, switchMap, catchError, EMPTY, startWith } from 'rxjs';

import { StatusBadgeComponent } from '../../../shared/components/status-badge/status-badge.component';
import { UploadZoneComponent } from './components/upload-zone/upload-zone.component';
import { ExtractionPreviewComponent } from './components/extraction-preview/extraction-preview.component';
import { DocumentRecord } from './models/document.model';
import { DocumentsService, DEFAULT_COURSE_ID, SyncResult } from './services/documents.service';

type TerminalStatus = 'ready' | 'failed';
type SyncState = 'idle' | 'syncing' | 'done' | 'error';

@Component({
  selector: 'app-materials',
  standalone: true,
  imports: [StatusBadgeComponent, UploadZoneComponent, ExtractionPreviewComponent],
  templateUrl: './materials.component.html',
  styleUrl: './materials.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MaterialsComponent implements OnInit {
  private readonly svc        = inject(DocumentsService);
  private readonly destroyRef = inject(DestroyRef);

  readonly documents     = signal<DocumentRecord[]>([]);
  readonly selectedDocId = signal<string | null>(null);
  readonly isLoading     = signal(true);
  readonly listError     = signal<string | null>(null);
  readonly syncState     = signal<SyncState>('idle');
  readonly syncResult    = signal<SyncResult | null>(null);
  readonly syncError     = signal<string | null>(null);

  readonly selectedDoc = computed(() =>
    this.documents().find(d => d.document_id === this.selectedDocId()) ?? null
  );

  ngOnInit(): void {
    // Poll the full list every 4 s; stops when component is destroyed.
    interval(4000).pipe(
      startWith(0),
      takeUntilDestroyed(this.destroyRef),
      switchMap(() => this.svc.list(DEFAULT_COURSE_ID).pipe(
        catchError(() => {
          this.listError.set('Could not reach the backend. Is the server running?');
          return EMPTY;
        })
      )),
    ).subscribe(docs => {
      this.isLoading.set(false);
      this.listError.set(null);
      this.mergeList(docs);
    });
  }

  onFilesSelected(files: File[]): void {
    for (const file of files) this.startUpload(file);
  }

  private startUpload(file: File): void {
    const tempId = `upload-${Date.now()}-${Math.random()}`;
    const placeholder: DocumentRecord = {
      document_id: tempId,
      filename: file.name,
      status: 'pending',
      created_at: new Date().toISOString(),
      error_msg: null,
      uploadProgress: 0,
    };
    this.documents.update(list => [placeholder, ...list]);

    this.svc.upload(file, DEFAULT_COURSE_ID).pipe(
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: event => {
        if (event.type === 'progress') {
          this.updateDoc(tempId, { uploadProgress: event.percent });
        } else if (event.type === 'complete') {
          const real = event.response;
          this.updateDoc(tempId, {
            document_id: real.document_id,
            status: real.status,
            uploadProgress: undefined,
          });
        }
      },
      error: () => {
        this.updateDoc(tempId, { status: 'failed', error_msg: 'Upload failed.', uploadProgress: undefined });
      },
    });
  }

  syncToLocal(): void {
    if (this.syncState() === 'syncing') return;
    this.syncState.set('syncing');
    this.syncResult.set(null);
    this.syncError.set(null);

    this.svc.sync(DEFAULT_COURSE_ID).pipe(
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: result => {
        this.syncResult.set(result);
        this.syncState.set('done');
      },
      error: () => {
        this.syncError.set('Sync failed. Is the ETL backend reachable?');
        this.syncState.set('error');
      },
    });
  }

  selectDoc(docId: string): void {
    this.selectedDocId.update(cur => cur === docId ? null : docId);
  }

  private mergeList(fresh: DocumentRecord[]): void {
    this.documents.update(current => {
      const inFlight = current.filter(d => d.uploadProgress !== undefined);
      const freshIds = new Set(fresh.map(d => d.document_id));
      const preserved = inFlight.filter(d => !freshIds.has(d.document_id));
      return [...preserved, ...fresh];
    });
  }

  private updateDoc(id: string, patch: Partial<DocumentRecord>): void {
    this.documents.update(list =>
      list.map(d => d.document_id === id ? { ...d, ...patch } : d)
    );
  }

  progressWidth(doc: DocumentRecord): string {
    if (doc.uploadProgress !== undefined) return `${doc.uploadProgress}%`;
    if (doc.status === 'ready')      return '100%';
    if (doc.status === 'processing') return '60%';
    return '0%';
  }

  formatDate(iso: string | null): string {
    if (!iso) return '';
    const d = new Date(iso);
    const diff = Date.now() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1)  return 'Just now';
    if (mins < 60) return `${mins} min ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24)  return `${hrs} h ago`;
    return d.toLocaleDateString();
  }

  fileExt(filename: string): string {
    return filename.split('.').pop()?.toUpperCase() ?? 'FILE';
  }

  isTerminal(status: string): status is TerminalStatus {
    return status === 'ready' || status === 'failed';
  }
}
