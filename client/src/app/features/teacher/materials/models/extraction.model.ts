export interface Chunk {
  chunk_id: string;
  page_number: number | null;
  text: string;
}

export interface ImageRecord {
  image_id: string;
  /** Presigned or proxied URL — null until backend serves it */
  url: string | null;
  vlm_description: string | null;
  page_number: number | null;
}

export interface KgNode {
  label: string;
  type: 'primary' | 'secondary';
}

export interface KgEdge {
  predicate: string;
}

export type KgElement = KgNode | KgEdge;

export interface ExtractionData {
  chunks: Chunk[];
  images: ImageRecord[];
  kg: KgElement[];
}
