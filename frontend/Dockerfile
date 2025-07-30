# 1. Build stage
FROM node:20-alpine AS build
WORKDIR /app

# Copy dependencies files first (for better Docker cache use)
COPY package*.json ./
RUN npm ci

# Copy the rest of your code
COPY . .

# Build the Vite app for production
RUN npm run build

# 2. Serve stage: Use Nginx or a simple static server
FROM nginx:stable-alpine

# Copy build output to Nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html

# (Optional) Nginx config for single-page apps
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
