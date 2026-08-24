# Painel do robô

Tela estática que mostra o retrato do dia: patrimônio, posições, ordens com o
motivo de cada uma, e o estado da cadeia de hash do livro-razão.

## Por que é estático

O painel não fala com banco nenhum e não roda nada em servidor. O robô produz
`relatorios/painel.json` depois de verificar a integridade da cadeia, e o build
transforma esse JSON em HTML.

Três consequências, todas boas:

- **Sem chave em runtime.** Não existe credencial para vazar porque não existe
  chamada a fazer.
- **Sem conta mensal.** Hospedar HTML é gratuito em qualquer lugar e continua
  sendo daqui a vinte anos. Um painel que exige servidor exige alguém pagando
  o servidor.
- **Sem número não auditado.** A tela só consegue mostrar o que passou pela
  verificação, porque é a única fonte que ela tem.

## Rodar localmente

```bash
# 1. o robô gera o retrato (na raiz do repositório)
python scripts/rodar_diario.py

# 2. o painel lê esse retrato
cd painel
npm install
npm run dev        # http://localhost:3000
```

`npm run build` gera a versão estática em `out/`.

## Publicar na Vercel

O projeto é privado e deve continuar privado. Um extrato de patrimônio não tem
por que estar acessível na internet aberta.

1. Importe o repositório na Vercel e aponte o **Root Directory** para `painel`.
2. Em **Settings → Deployment Protection**, ligue **Vercel Authentication**.
   Só quem estiver logado na sua conta abre a página.
3. Em **Settings → Git**, desligue o deploy automático a cada push: quem
   dispara o deploy é o robô, depois de auditar.
4. Crie um **Deploy Hook** e guarde a URL como segredo `VERCEL_DEPLOY_HOOK` no
   GitHub. O workflow `robo-diario.yml` chama esse gancho ao fim de cada
   pregão.

Os cabeçalhos de segurança (CSP restritiva, `noindex`, `frame-ancestors none`,
HSTS) estão em `vercel.json` e valem para todas as rotas.

## Se o build falhar por falta de dados

`scripts/preparar-dados.mjs` derruba o build quando `relatorios/painel.json`
não existe. É de propósito: publicar uma tela vazia com aparência de
funcionando é pior que não publicar.
