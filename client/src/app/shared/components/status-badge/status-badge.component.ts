import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { DocumentStatus } from '../../../features/teacher/materials/models/document.model';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  template: `
    <span class="badge" [class]="'badge--' + status()">
      <span class="badge__dot"></span>
      {{ label() }}
    </span>
  `,
  styleUrl: './status-badge.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StatusBadgeComponent {
  readonly status = input.required<DocumentStatus>();

  get label() {
    return () => {
      const map: Record<DocumentStatus, string> = {
        pending:    'Pending',
        processing: 'Processing',
        ready:      'Ready',
        failed:     'Failed',
      };
      return map[this.status()];
    };
  }
}
