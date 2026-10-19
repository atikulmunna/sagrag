# Static web UI behind nginx (non-root image, listens on 8080).
FROM nginxinc/nginx-unprivileged:1.29-alpine

COPY infra/nginx/web.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
