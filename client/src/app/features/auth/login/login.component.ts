import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
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
  readonly selectedRole = signal<UserRole | null>(null);

  selectRole(role: UserRole): void { this.selectedRole.set(role); }
  clearRole():                void { this.selectedRole.set(null); }

  onLoginSuccess(credentials: LoginCredentials): void {
    // TODO: navigate to role-specific dashboard
    console.log('[Login] authenticated as:', credentials.role);
  }
}
