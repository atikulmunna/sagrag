{{- define "sag-rag.name" -}}
sag-rag
{{- end }}

{{- define "sag-rag.fullname" -}}
{{ include "sag-rag.name" . }}
{{- end }}
