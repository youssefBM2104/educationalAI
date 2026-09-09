import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { AuthFormComponent } from '../components/auth-form/auth-form.component';
import { RolePanelComponent } from '../components/role-panel/role-panel.component';
import { LoginCredentials, UserRole } from '../auth.types';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [RolePanelComponent, AuthFormComponent],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LoginComponent {
  private readonly router = inject(Router);

  readonly selectedRole = signal<UserRole | null>(null);

  selectRole(role: UserRole): void { this.selectedRole.set(role); }
  clearRole():                void { this.selectedRole.set(null); }

  onLoginSuccess(credentials: LoginCredentials): void {
    // TODO: route to student dashboard when that page is built
    const destination = credentials.role === 'lecturer' ? '/teacher' : '/teacher';
    this.router.navigate([destination]);
  }
}
