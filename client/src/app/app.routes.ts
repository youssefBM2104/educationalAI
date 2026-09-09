import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    loadComponent: () =>
      import('./features/auth/login/login.component').then(m => m.LoginComponent),
  },
  {
    path: 'teacher',
    loadComponent: () =>
      import('./features/teacher/dashboard/dashboard.component').then(m => m.DashboardComponent),
    loadChildren: () =>
      import('./features/teacher/teacher.routes').then(m => m.TEACHER_ROUTES),
  },
  { path: '**', redirectTo: '' },
];
