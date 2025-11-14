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

# Inject runtime API base via small config
ARG API_BASE=/api
RUN echo "window.VITE_API_BASE='${API_BASE}';" > /usr/share/nginx/html/config.js

EXPOSE 80
