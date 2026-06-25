# SONA — Foto → The Sims 3

Recebe a **foto de um rosto humano real** e devolve um **retrato no estilo de
The Sims 3**, preservando a identidade da pessoa. Backend modular pensado para ser
consumido por um app **mobile** (o celular só envia a foto; a IA roda no servidor).

> Arquitetura, justificativas técnicas e como cada requisito é atendido:
> ver [`ARQUITETURA.md`](./ARQUITETURA.md).

## Fluxo

```
foto → [A] pré-processo → [B] validação de rosto (porteiro) →
       [C] geração InstantID (estilo Sims 3) → [D] verificação de fidelidade → resultado
```

- **B (porteiro):** detecção (InsightFace) + "é humano real?" (CLIP zero-shot) +
  estratégia de múltiplos rostos. Rejeita desenho, render 3D, personagem de jogo,
  animal, objeto e paisagem.
- **C (geração):** InstantID sobre SDXL — preserva identidade e pose a partir de
  **uma única foto**, mudando apenas o estilo artístico.
- **D (fidelidade):** compara identidade (cosseno ArcFace) entre entrada e saída e
  re-tenta se ficar abaixo do limiar.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # preencha REPLICATE_API_TOKEN
```

A primeira execução baixa os modelos do InsightFace (`buffalo_l`) e do CLIP.

## Rodar

```bash
uvicorn app.main:app --reload
# UI de teste:  http://localhost:8000/
# Swagger:      http://localhost:8000/docs
```

### Exemplo de chamada

```bash
curl -F "file=@foto.jpg" http://localhost:8000/transform
```

Resposta (sucesso): PNG em base64 + `similarity` (fidelidade de identidade),
`attempts`, `provider`. Imagem rejeitada → HTTP **422** com `reason` e `message`.

## Testes

```bash
pytest -q     # rodam sem GPU: a IA pesada é substituída por dublês
```

Os testes cobrem todas as regras do porteiro (sem rosto, baixa confiança, rosto
pequeno, não-humano, múltiplos rostos, rosto principal) e o pipeline (rejeição,
sucesso, loop de re-tentativa por baixa fidelidade) e a fiação da API.

## Configuração

Todos os limiares são ajustáveis por variável de ambiente — ver `.env.example`.
Trocar `SONA_PROVIDER=replicate` por `comfyui` permite self-host numa GPU própria
(custo zero por imagem) sem mudar o resto do código.

## Integração mobile

O app chama `POST /transform` (multipart `file`). Recomendado adicionar atrás de um
gateway com autenticação e fila de jobs para escala. Para LoRA própria de estilo
The Sims 3, defina `SONA_STYLE_LORA`.
