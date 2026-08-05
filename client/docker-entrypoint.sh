#!/bin/sh
cat > /usr/share/nginx/html/env.js <<ENVEOF
window.__env = {
  apiUrl: "${API_URL:-http://localhost:8420}"
};
ENVEOF
