export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface DocumentRecord {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  created_at: string | null;
  error_msg: string | null;
  /** Upload HTTP progress (0–100). Only present while the HTTP transfer is in flight. */
  uploadProgress?: number;
}

export interface UploadResponse {
  document_id: string;
  status: DocumentStatus;
  /** Returned on dedup: the document was already indexed. */
  message?: string;
}

export interface StatusResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  error_msg: string | null;
}
