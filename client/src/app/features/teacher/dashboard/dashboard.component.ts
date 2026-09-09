import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

interface NavItem {
  label: string;
  icon: string;
  route: string | null;
  disabled: boolean;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardComponent {
  readonly sidebarOpen = signal(true);

  readonly workspaceNav: NavItem[] = [
    { label: 'Materials',       icon: 'doc',   route: 'materials', disabled: false },
    { label: 'Question bank',   icon: 'pencil', route: null,        disabled: true  },
    { label: 'Exam generation', icon: 'flask',  route: null,        disabled: true  },
    { label: 'Slide generation',icon: 'slides', route: null,        disabled: true  },
  ];

  readonly accountNav: NavItem[] = [
    { label: 'Settings', icon: 'gear', route: null, disabled: true },
  ];

  toggleSidebar(): void { this.sidebarOpen.update(o => !o); }

  onDisabledClick(event: Event): void { event.preventDefault(); }
}
