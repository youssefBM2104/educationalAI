import {
  ChangeDetectionStrategy,
  Component,
  inject,
  input,
  OnInit,
  signal,
} from '@angular/core';
import { DocumentRecord } from '../../models/document.model';
import { ExtractionData, ImageRecord } from '../../models/extraction.model';
import { DocumentsService } from '../../services/documents.service';
import { KgGraphComponent } from '../kg-graph/kg-graph.component';
import { environment } from '../../../../../../environments/environment';
import { catchError, EMPTY } from 'rxjs';

type Tab = 'chunks' | 'images' | 'kg';

@Component({
  selector: 'app-extraction-preview',
  standalone: true,
  imports: [KgGraphComponent],
  templateUrl: './extraction-preview.component.html',
  styleUrl: './extraction-preview.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExtractionPreviewComponent implements OnInit {
  readonly document = input.required<DocumentRecord>();

  readonly activeTab = signal<Tab>('chunks');
  readonly data = signal<ExtractionData | null>(null);
  readonly isLoading = signal(true);
  readonly error = signal<string | null>(null);
  readonly selectedImage = signal<ImageRecord | null>(null);

  private readonly svc = inject(DocumentsService);

  ngOnInit(): void {
    this.svc
      .getExtraction(this.document().document_id)
      .pipe(
        catchError(() => {
          this.error.set('Could not load extraction data.');
          this.isLoading.set(false);
          return EMPTY;
        })
      )
      .subscribe(result => {
        this.data.set(result);
        this.isLoading.set(false);
      });
  }

  get conceptCount(): number {
    return (this.data()?.kg ?? []).filter(el => 'label' in el).length;
  }

  setTab(tab: Tab): void { this.activeTab.set(tab); }

  selectImage(img: ImageRecord): void { this.selectedImage.set(img); }
  closeImage(): void { this.selectedImage.set(null); }

  truncate(text: string | null, limit = 120): string {
    if (!text) return '';
    return text.length > limit ? text.slice(0, limit) + '…' : text;
  }

  imageUrl(path: string | null): string | null {
    if (!path) return null;
    return path.startsWith('http') ? path : `${environment.apiUrl}${path}`;
  }
}
