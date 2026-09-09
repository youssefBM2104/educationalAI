import { Routes } from '@angular/router';

export const TEACHER_ROUTES: Routes = [
  { path: '', redirectTo: 'materials', pathMatch: 'full' },
  {
    path: 'materials',
    loadComponent: () =>
      import('./materials/materials.component').then(m => m.MaterialsComponent),
  },
];
