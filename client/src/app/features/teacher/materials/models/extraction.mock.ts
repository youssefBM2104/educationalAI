/**
 * DEV MOCK — used until GET /documents/{id}/extraction exists on the backend.
 * Replace the ExtractionPreviewComponent's data source once that endpoint ships.
 */
import { ExtractionData } from './extraction.model';

export const MOCK_EXTRACTION: ExtractionData = {
  chunks: [
    {
      chunk_id: 'c1',
      page_number: 2,
      text: 'Price elasticity of demand measures how the quantity demanded responds to a change in price. Goods with elasticity greater than 1 are considered elastic — a small price change causes a proportionally larger change in quantity demanded.',
    },
    {
      chunk_id: 'c2',
      page_number: 2,
      text: 'Inelastic goods, by contrast, see relatively little change in demand despite price fluctuations. Classic examples include insulin and other necessity medications, where substitution options are limited.',
    },
    {
      chunk_id: 'c3',
      page_number: 4,
      text: 'The cross-price elasticity of demand captures how the quantity demanded of one good responds to a price change in another good. A positive value indicates substitute goods; a negative value indicates complements.',
    },
  ],
  images: [
    {
      image_id: 'img1',
      url: null,
      vlm_description: 'Line chart showing the demand curve shifting left as the price of a substitute good decreases, with annotated equilibrium points.',
      page_number: 2,
    },
    {
      image_id: 'img2',
      url: null,
      vlm_description: 'Table comparing price elasticity coefficients across five common goods: bread, insulin, luxury cars, petrol, and smartphones.',
      page_number: 3,
    },
    {
      image_id: 'img3',
      url: null,
      vlm_description: 'Diagram contrasting elastic vs. inelastic demand curve steepness, with slope labels and example goods annotated.',
      page_number: 5,
    },
  ],
  kg: [
    { label: 'Price Elasticity', type: 'primary' },
    { predicate: 'influences' },
    { label: 'Demand Curve', type: 'secondary' },
    { predicate: 'shifts with' },
    { label: 'Substitute Goods', type: 'primary' },
    { predicate: 'causes' },
    { label: 'Inelastic Demand', type: 'secondary' },
  ],
};
