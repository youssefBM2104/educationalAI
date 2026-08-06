import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  input,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import NeoVis, { NeovisConfig } from 'neovis.js';
import { environment } from '../../../../../../environments/environment';

@Component({
  selector: 'app-kg-graph',
  standalone: true,
  template: `<div #container class="kg-graph-container"></div>`,
  styles: [`
    .kg-graph-container {
      width: 100%;
      height: 480px;
      border-radius: 10px;
      overflow: hidden;
      background: #f6f7f9;
    }
  `],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KgGraphComponent implements AfterViewInit, OnDestroy {
  readonly documentId = input.required<string>();

  @ViewChild('container', { static: true }) container!: ElementRef<HTMLDivElement>;

  private viz: NeoVis | null = null;

  ngAfterViewInit(): void {
    const config: NeovisConfig = {
      containerId: this.container.nativeElement.id || this._ensureId(),
      neo4j: {
        serverUrl:      environment.neo4j.url,
        serverUser:     environment.neo4j.user,
        serverPassword: environment.neo4j.password,
      },
      visConfig: {
        nodes: { shape: 'dot', size: 16, font: { size: 14 } },
        edges: { arrows: { to: { enabled: true } }, font: { size: 11 } },
        physics: { stabilization: { iterations: 100 } },
      },
      labels: {
        Concept:  { label: 'id', color: '#0E7C72' },
        Formula:  { label: 'id', color: '#1B3A6B' },
        Theorem:  { label: 'id', color: '#7B5EA7' },
        Example:  { label: 'id', color: '#A5670F' },
        Method:   { label: 'id', color: '#2E7D32' },
      },
      relationships: {
        PREREQUISITE: { label: 'PREREQUISITE' },
        EXTENDS:      { label: 'EXTENDS' },
        DEFINES:      { label: 'DEFINES' },
        APPLIES_TO:   { label: 'APPLIES_TO' },
        ILLUSTRATES:  { label: 'ILLUSTRATES' },
        PART_OF:      { label: 'PART_OF' },
      },
      initialCypher: `
        MATCH (d:Document)-[:MENTIONS]->(n)
        WHERE d.document_id = '${this.documentId()}'
        WITH collect(DISTINCT n) AS nodes
        UNWIND nodes AS a
        OPTIONAL MATCH (a)-[r]->(b) WHERE b IN nodes
        RETURN a, r, b
      `,
    };

    this.viz = new NeoVis(config);
    this.viz.render();
  }

  ngOnDestroy(): void {
    this.viz?.clearNetwork();
  }

  private _ensureId(): string {
    const id = `kg-graph-${Math.random().toString(36).slice(2)}`;
    this.container.nativeElement.id = id;
    return id;
  }
}
