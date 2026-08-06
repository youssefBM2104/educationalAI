#!/bin/sh
cat > /usr/share/nginx/html/env.js <<ENVEOF
window.__env = {
  apiUrl:       "${API_URL:-http://localhost:8420}",
  neo4jUrl:     "${NEO4J_URL:-bolt://localhost:7687}",
  neo4jUser:    "${NEO4J_USER:-neo4j}",
  neo4jPassword:"${NEO4J_PASSWORD:-}"
};
ENVEOF
