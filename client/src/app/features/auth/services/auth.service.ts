import { Injectable } from '@angular/core';
import { Observable, of, delay } from 'rxjs';
import { AuthResult, LoginCredentials } from '../auth.types';

@Injectable({ providedIn: 'root' })
export class AuthService {
  // Stub — replace with real HTTP POST /api/auth/login
  login(credentials: LoginCredentials): Observable<AuthResult> {
    console.log('[AuthService stub] login:', credentials.role, credentials.email);
    return of({ success: true }).pipe(delay(900));
  }
}
