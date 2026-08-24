/** @type {import('next').NextConfig} */
const nextConfig = {
  // Exportação estática: o painel é um retrato do dia, gerado pelo robô e
  // publicado como HTML. Sem servidor, sem banco exposto, sem chave em
  // runtime — e por isso sem superfície de ataque e sem conta mensal.
  output: "export",
  reactStrictMode: true,
  poweredByHeader: false,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
