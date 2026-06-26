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

## Estilo The Sims: providers e LoRA

A geracao usa um `provider` selecionavel em `SONA_PROVIDER`:

- `flux_pulid` (padrao) - Flux + PuLID via `bytedance/flux-pulid`. Preserva
  identidade a partir de uma foto e aplica o estilo The Sims pelo prompt
  (gatilho `THSMS`). Funciona imediatamente com o token do Replicate, sem GPU.
- `comfyui_replicate` - Flux + PuLID + a LoRA Sims (`dvyio/flux-lora-the-sims`)
  num workflow ComfyUI rodando via `fofr/any-comfyui-workflow`. Maxima fidelidade
  de estilo. Requer validar o workflow `workflows/flux_pulid_lora.json` no seu
  ComfyUI (Save -> API Format), pois os nos PuLID-Flux dependem do pack instalado.
- `replicate` - SDXL InstantID (fallback).
- `comfyui` - ComfyUI self-hosted (GPU propria).

LoRA usada: `dvyio/flux-lora-the-sims` (base Flux.1-dev, gatilho
`video game screenshot in the style of THSMS`).

Licenca: FLUX.1-dev e essa LoRA sao de uso nao-comercial. Para um produto
comercial, migre para FLUX.1-schnell (Apache), volte ao SDXL/InstantID, ou treine
uma LoRA propria.
