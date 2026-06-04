# Agenda os ciclos do QuantBot no Agendador de Tarefas do Windows.
# Rode UMA vez, em PowerShell, dentro da pasta do projeto:
#     powershell -ExecutionPolicy Bypass -File scripts\agendar_windows.ps1
#
# Cria 3 tarefas:
#   - QuantBot-Diario  : dias úteis 18:30 (após fechamento da B3)
#   - QuantBot-Semanal : segunda-feira 08:00 (screening + recomendações)
#   - QuantBot-Mensal  : dia 1 às 09:00 (consolidação + projeção)
#
# Para remover depois:
#   Unregister-ScheduledTask -TaskName 'QuantBot-Diario','QuantBot-Semanal','QuantBot-Mensal' -Confirm:$false

$ErrorActionPreference = "Stop"

$projectDir = (Resolve-Path "$PSScriptRoot\..").Path

# Usa o python da venv se existir, senão o python do PATH.
$venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = (Get-Command python).Source
}

Write-Host "Projeto: $projectDir"
Write-Host "Python : $python"

function New-QuantBotTask {
    param($Name, $Args, $Trigger)
    $action = New-ScheduledTaskAction -Execute $python `
        -Argument "main_income.py $Args" -WorkingDirectory $projectDir
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger `
        -Settings $settings -Description "QuantBot ciclo automatico" -Force | Out-Null
    Write-Host "  [OK] Tarefa criada: $Name"
}

# Diário — dias úteis 18:30
$tDaily = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 6:30PM
New-QuantBotTask -Name "QuantBot-Diario" -Args "--daily" -Trigger $tDaily

# Semanal — segunda 08:00
$tWeekly = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 8:00AM
New-QuantBotTask -Name "QuantBot-Semanal" -Args "--weekly" -Trigger $tWeekly

# Mensal — dia 1 às 09:00 (schtasks suporta gatilho mensal real)
# Envolve em cmd /c para entrar na pasta do projeto (paths relativos).
$monthlyCmd = "cmd /c cd /d `"$projectDir`" && `"$python`" main_income.py --monthly"
schtasks /Create /TN "QuantBot-Mensal" /TR $monthlyCmd /SC MONTHLY /D 1 /ST 09:00 /F | Out-Null
Write-Host "  [OK] Tarefa criada: QuantBot-Mensal"

Write-Host ""
Write-Host "Pronto. As 3 tarefas estao agendadas. Veja em 'Agendador de Tarefas'."
Write-Host "Teste manual agora com:  $python main_income.py --status"
