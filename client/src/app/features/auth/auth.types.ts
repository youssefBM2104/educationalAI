export type UserRole = 'lecturer' | 'student';

export interface LoginCredentials {
  email: string;
  password: string;
  role: UserRole;
}

export interface AuthResult {
  success: boolean;
  message?: string;
}
