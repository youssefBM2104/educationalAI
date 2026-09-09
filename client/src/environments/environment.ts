export const environment = {
  production: false,
  apiUrl: (window as any).__env?.apiUrl || 'http://localhost:8420',
  neo4j: {
    url:      (window as any).__env?.neo4jUrl      || 'bolt://localhost:7687',
    user:     (window as any).__env?.neo4jUser     || 'neo4j',
    password: (window as any).__env?.neo4jPassword || '',
  },
};
