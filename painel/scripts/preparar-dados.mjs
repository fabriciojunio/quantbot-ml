/**
 * Copia o retrato gerado pelo robô para dentro do build.
 *
 * O painel nunca lê o banco de auditoria direto. Ele lê um JSON que o robô
 * produziu depois de conferir a integridade da cadeia — o que significa que a
 * tela não tem como mostrar um número que não passou pela verificação.
 *
 * Se o arquivo não existir, o build falha em vez de publicar uma tela vazia
 * com aparência de funcionando.
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const aqui = dirname(fileURLToPath(import.meta.url));
const origem = resolve(aqui, "..", "..", "relatorios", "painel.json");
const destino = resolve(aqui, "..", "dados", "painel.json");

if (!existsSync(origem)) {
  console.error(
    `\n  Não achei ${origem}.\n` +
      `  Rode antes:  python scripts/rodar_diario.py\n`
  );
  process.exit(1);
}

mkdirSync(dirname(destino), { recursive: true });
copyFileSync(origem, destino);
console.log(`  Dados do painel copiados de ${origem}`);
