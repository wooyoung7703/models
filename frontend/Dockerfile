FROM node:20-alpine AS build
WORKDIR /app

# Install deps
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Copy source and build
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

# Nginx conf to serve SPA and proxy /api to backend
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Static assets
COPY --from=build /app/dist /usr/share/nginx/html

# Inject runtime API base + admin token into a small config
ARG API_BASE=/api
ARG WS_ADMIN_TOKEN=""
RUN printf "window.VITE_API_BASE='%s';\nwindow.VITE_WS_URL='ws://'+(window.location.hostname||'localhost')+':8022';\nwindow.WS_ONLY=true;\nwindow.WS_ADMIN_TOKEN='%s';\n" "${API_BASE}" "${WS_ADMIN_TOKEN}" > /usr/share/nginx/html/config.js

EXPOSE 80
