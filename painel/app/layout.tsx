import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Livro do Robô",
  description:
    "Patrimônio, decisões e integridade do livro-razão. Preços reais, dinheiro simulado.",
  robots: {
    // Um extrato de patrimônio não tem por que estar em índice de busca.
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body>
        <div className="folha">{children}</div>
      </body>
    </html>
  );
}
