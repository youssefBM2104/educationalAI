import { ChangeDetectionStrategy, Component, output, signal } from '@angular/core';

const ACCEPTED = '.pdf,.docx,.pptx';
const ACCEPTED_MIME = ['application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation'];

@Component({
  selector: 'app-upload-zone',
  standalone: true,
  templateUrl: './upload-zone.component.html',
  styleUrl: './upload-zone.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class UploadZoneComponent {
  readonly filesSelected = output<File[]>();

  readonly isDragging = signal(false);

  readonly accept = ACCEPTED;

  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.isDragging.set(true);
  }

  onDragLeave(e: DragEvent): void {
    e.preventDefault();
    this.isDragging.set(false);
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.isDragging.set(false);
    const files = Array.from(e.dataTransfer?.files ?? []).filter(f =>
      ACCEPTED_MIME.includes(f.type)
    );
    if (files.length) this.filesSelected.emit(files);
  }

  onInputChange(e: Event): void {
    const input = e.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    if (files.length) this.filesSelected.emit(files);
    input.value = '';
  }

  triggerInput(input: HTMLInputElement): void {
    input.click();
  }
}
