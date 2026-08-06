import { ChangeDetectionStrategy, Component, input, signal } from '@angular/core';
import { DocumentRecord } from '../../models/document.model';
import { ExtractionData, KgEdge, KgElement, KgNode } from '../../models/extraction.model';
import { MOCK_EXTRACTION } from '../../models/extraction.mock';

type Tab = 'chunks' | 'images' | 'kg';

@Component({
  selector: 'app-extraction-preview',
  standalone: true,
  templateUrl: './extraction-preview.component.html',
  styleUrl: './extraction-preview.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExtractionPreviewComponent {
  readonly document = input.required<DocumentRecord>();

  readonly activeTab = signal<Tab>('chunks');

  /**
   * DEV — served from mock until GET /documents/{id}/extraction exists.
   * Once the endpoint ships, inject DocumentsService and call it here,
   * keyed on document().document_id.
   */
  readonly data: ExtractionData = MOCK_EXTRACTION;
  readonly isMock = true;

  get conceptCount(): number {
    return this.data.kg.filter(el => 'label' in el).length;
  }

  setTab(tab: Tab): void { this.activeTab.set(tab); }

  isNode(el: KgElement): el is KgNode { return 'label' in el; }
  isEdge(el: KgElement): el is KgEdge { return 'predicate' in el; }
}
