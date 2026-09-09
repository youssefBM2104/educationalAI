import { ChangeDetectionStrategy, Component, inject, input, output, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { LoginCredentials, UserRole } from '../../auth.types';

@Component({
  selector: 'app-auth-form',
  standalone: true,
  imports: [ReactiveFormsModule],
  templateUrl: './auth-form.component.html',
  styleUrl: './auth-form.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AuthFormComponent {
  readonly role = input.required<UserRole>();

  readonly submitted = output<LoginCredentials>();
  readonly back      = output<void>();

  private readonly fb          = inject(FormBuilder);
  private readonly authService = inject(AuthService);

  readonly isLoading  = signal(false);
  readonly loginError = signal<string | null>(null);

  readonly form = this.fb.group({
    email:    ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(6)]],
  });

  get emailCtrl()    { return this.form.get('email')!; }
  get passwordCtrl() { return this.form.get('password')!; }

  get emailError(): string | null {
    const c = this.emailCtrl;
    if (!c.dirty && !c.touched) return null;
    if (c.hasError('required')) return 'Email is required.';
    if (c.hasError('email'))    return 'Enter a valid email address.';
    return null;
  }

  get passwordError(): string | null {
    const c = this.passwordCtrl;
    if (!c.dirty && !c.touched) return null;
    if (c.hasError('required'))   return 'Password is required.';
    if (c.hasError('minlength'))  return 'Password must be at least 6 characters.';
    return null;
  }

  onSubmit(): void {
    if (this.form.invalid) { this.form.markAllAsTouched(); return; }

    this.isLoading.set(true);
    this.loginError.set(null);

    const credentials: LoginCredentials = {
      email:    this.form.value.email!,
      password: this.form.value.password!,
      role:     this.role(),
    };

    this.authService.login(credentials).subscribe({
      next: (result) => {
        this.isLoading.set(false);
        if (result.success) {
          this.submitted.emit(credentials);
        } else {
          this.loginError.set(result.message ?? 'Login failed. Please try again.');
        }
      },
      error: () => {
        this.isLoading.set(false);
        this.loginError.set('Something went wrong. Please try again.');
      },
    });
  }

  goBack(): void { this.back.emit(); }
}
