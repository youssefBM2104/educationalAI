import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { UserRole } from '../../auth.types';

interface PanelConfig { label: string; tagline: string; cta: string }

const CONFIG: Record<UserRole, PanelConfig> = {
  lecturer: { label: 'Lecturer', tagline: 'Design lessons. Grade smarter.', cta: 'Enter as Lecturer' },
  student:  { label: 'Student',  tagline: 'Learn deeper. Practice faster.', cta: 'Enter as Student'  },
};

@Component({
  selector: 'app-role-panel',
  standalone: true,
  templateUrl: './role-panel.component.html',
  styleUrl: './role-panel.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    '[class.role-panel--lecturer]': "role() === 'lecturer'",
    '[class.role-panel--student]':  "role() === 'student'",
    '[class.role-panel--selected]': 'isSelected()',
    '[class.role-panel--receded]':  'isReceded',
    '[attr.role]':                  "isSelected() ? null : 'button'",
    '[attr.tabindex]':              "isSelected() ? '-1' : '0'",
    '[attr.aria-pressed]':          'isSelected()',
    '[attr.aria-label]':            "isSelected() ? null : config.cta",
    '(click)':                      'handleClick()',
    '(keydown.enter)':              'handleClick()',
    '(keydown.space)':              'handleKeySpace($event)',
  },
})
export class RolePanelComponent {
  readonly role          = input.required<UserRole>();
  readonly isSelected    = input(false);
  readonly otherSelected = input(false);
  readonly roleSelected  = output<void>();

  get config(): PanelConfig { return CONFIG[this.role()]; }
  get isReceded(): boolean  { return this.otherSelected() && !this.isSelected(); }

  handleClick(): void {
    if (!this.isSelected()) this.roleSelected.emit();
  }

  handleKeySpace(event: Event): void {
    event.preventDefault();
    this.handleClick();
  }
}
