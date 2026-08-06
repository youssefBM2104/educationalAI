import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpEventType, HttpRequest } from '@angular/common/http';
import { Observable, filter, map } from 'rxjs';
import { environment } from '../../../../../environments/environment';
import { DocumentRecord, StatusResponse, UploadResponse } from '../models/document.model';

export interface UploadProgress {
  type: 'progress';
  percent: number;
}
export interface UploadComplete {
  type: 'complete';
  response: UploadResponse;
}
export type UploadEvent = UploadProgress | UploadComplete;

/** Hard-coded for now — add course-selection UI when needed. */
export const DEFAULT_COURSE_ID = 'default';

@Injectable({ providedIn: 'root' })
export class DocumentsService {
  private readonly http = inject(HttpClient);
  private readonly base = `${environment.apiUrl}/documents`;

  list(courseId = DEFAULT_COURSE_ID): Observable<DocumentRecord[]> {
    return this.http.get<DocumentRecord[]>(`${this.base}/?course_id=${courseId}`);
  }

  upload(file: File, courseId = DEFAULT_COURSE_ID): Observable<UploadEvent> {
    const fd = new FormData();
    fd.append('course_id', courseId);
    fd.append('file', file);

    const req = new HttpRequest('POST', `${this.base}/upload`, fd, {
      reportProgress: true,
    });

    return this.http.request<UploadResponse>(req).pipe(
      filter(e =>
        e.type === HttpEventType.UploadProgress ||
        e.type === HttpEventType.Response
      ),
      map(e => {
        if (e.type === HttpEventType.UploadProgress) {
          const percent = e.total ? Math.round((100 * e.loaded) / e.total) : 0;
          return { type: 'progress', percent } as UploadProgress;
        }
        if (e.type === HttpEventType.Response) {
          return { type: 'complete', response: e.body! } as UploadComplete;
        }
        throw new Error('unexpected event');
      })
    );
  }

  getStatus(documentId: string): Observable<StatusResponse> {
    return this.http.get<StatusResponse>(`${this.base}/${documentId}/status`);
  }
}
