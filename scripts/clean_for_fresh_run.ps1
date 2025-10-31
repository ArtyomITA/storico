# ============================================================================
# Script di pulizia per reset completo del progetto
# Rimuove tutti i file generati, mantenendo solo i dati raw e il codice sorgente
# ============================================================================

$ErrorActionPreference = "Continue"
Write-Host "🧹 Avvio pulizia completa per fresh run..." -ForegroundColor Cyan
Write-Host ""

# ==================================================
# 1. DATA/CURATED - Tutti i parquet generati
# ==================================================
Write-Host "📂 Pulizia data/curated/*.parquet..." -ForegroundColor Yellow
$curatedFiles = Get-ChildItem -Path "data\curated" -Filter "*.parquet" -File -ErrorAction SilentlyContinue
if ($curatedFiles) {
    $curatedFiles | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.Name)"
        Remove-Item -Path $_.FullName -Force
    }
} else {
    Write-Host "   ✓ Nessun file parquet in data/curated" -ForegroundColor Gray
}

# Rimuovi anche JSON metadata
$curatedJson = Get-ChildItem -Path "data\curated" -Filter "*.json" -File -ErrorAction SilentlyContinue
if ($curatedJson) {
    $curatedJson | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.Name)"
        Remove-Item -Path $_.FullName -Force
    }
}

# Rimuovi CSV in curated
$curatedCsv = Get-ChildItem -Path "data\curated" -Filter "*.csv" -File -ErrorAction SilentlyContinue
if ($curatedCsv) {
    $curatedCsv | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.Name)"
        Remove-Item -Path $_.FullName -Force
    }
}

Write-Host ""

# ==================================================
# 2. MODELS - Tutti i modelli addestrati
# ==================================================
Write-Host "📂 Pulizia models/*..." -ForegroundColor Yellow
if (Test-Path "models") {
    # Rimuovi file singoli
    Get-ChildItem -Path "models" -File -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.Name)"
        Remove-Item -Path $_.FullName -Force
    }
    
    # Rimuovi cartella autogluon
    if (Test-Path "models\autogluon") {
        Write-Host "   ❌ Rimosso: models\autogluon\"
        Remove-Item -Path "models\autogluon" -Recurse -Force
    }
} else {
    Write-Host "   ✓ Cartella models non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 3. REPORTS - Report HTML generati
# ==================================================
Write-Host "📂 Pulizia reports/*.html..." -ForegroundColor Yellow
if (Test-Path "reports") {
    $reports = Get-ChildItem -Path "reports" -Filter "*.html" -File -ErrorAction SilentlyContinue
    if ($reports) {
        $reports | ForEach-Object {
            Write-Host "   ❌ Rimosso: $($_.Name)"
            Remove-Item -Path $_.FullName -Force
        }
    } else {
        Write-Host "   ✓ Nessun report HTML" -ForegroundColor Gray
    }
} else {
    Write-Host "   ✓ Cartella reports non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 4. LOGS - Log di esecuzione
# ==================================================
Write-Host "📂 Pulizia logs/*.jsonl..." -ForegroundColor Yellow
if (Test-Path "logs") {
    $logs = Get-ChildItem -Path "logs" -Filter "*.jsonl" -File -ErrorAction SilentlyContinue
    if ($logs) {
        $logs | ForEach-Object {
            Write-Host "   ❌ Rimosso: $($_.Name)"
            Remove-Item -Path $_.FullName -Force
        }
    } else {
        Write-Host "   ✓ Nessun log jsonl" -ForegroundColor Gray
    }
} else {
    Write-Host "   ✓ Cartella logs non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 5. CATBOOST_INFO - Cache CatBoost
# ==================================================
Write-Host "📂 Pulizia catboost_info/*..." -ForegroundColor Yellow
if (Test-Path "catboost_info") {
    Write-Host "   ❌ Rimosso: catboost_info\"
    Remove-Item -Path "catboost_info" -Recurse -Force
} else {
    Write-Host "   ✓ Cartella catboost_info non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 6. __pycache__ - Cache Python
# ==================================================
Write-Host "📂 Pulizia __pycache__ ricorsiva..." -ForegroundColor Yellow
$pycache = Get-ChildItem -Path . -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue
if ($pycache) {
    $pycache | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.FullName -replace [regex]::Escape($PWD.Path), '.')"
        Remove-Item -Path $_.FullName -Recurse -Force
    }
} else {
    Write-Host "   ✓ Nessuna cache __pycache__" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 7. .pytest_cache - Cache pytest
# ==================================================
Write-Host "📂 Pulizia .pytest_cache..." -ForegroundColor Yellow
if (Test-Path ".pytest_cache") {
    Write-Host "   ❌ Rimosso: .pytest_cache\"
    Remove-Item -Path ".pytest_cache" -Recurse -Force
} else {
    Write-Host "   ✓ Cartella .pytest_cache non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 8. README.prev.md - Backup temporaneo
# ==================================================
Write-Host "📂 Pulizia README.prev.md..." -ForegroundColor Yellow
if (Test-Path "README.prev.md") {
    Write-Host "   ❌ Rimosso: README.prev.md"
    Remove-Item -Path "README.prev.md" -Force
} else {
    Write-Host "   ✓ README.prev.md non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 9. _artifacts - File generati di documentazione
# ==================================================
Write-Host "📂 Pulizia _artifacts/*..." -ForegroundColor Yellow
if (Test-Path "_artifacts") {
    $artifacts = Get-ChildItem -Path "_artifacts" -File -ErrorAction SilentlyContinue
    if ($artifacts) {
        $artifacts | ForEach-Object {
            Write-Host "   ❌ Rimosso: $($_.Name)"
            Remove-Item -Path $_.FullName -Force
        }
    } else {
        Write-Host "   ✓ Nessun artifact" -ForegroundColor Gray
    }
} else {
    Write-Host "   ✓ Cartella _artifacts non esistente" -ForegroundColor Gray
}

Write-Host ""

# ==================================================
# 10. File temporanei vari
# ==================================================
Write-Host "📂 Pulizia file temporanei (*.tmp, *.log, *.bak)..." -ForegroundColor Yellow
$tempFiles = Get-ChildItem -Path . -Include "*.tmp","*.log","*.bak" -File -Recurse -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }
if ($tempFiles) {
    $tempFiles | ForEach-Object {
        Write-Host "   ❌ Rimosso: $($_.FullName -replace [regex]::Escape($PWD.Path), '.')"
        Remove-Item -Path $_.FullName -Force
    }
} else {
    Write-Host "   ✓ Nessun file temporaneo" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ PULIZIA COMPLETATA" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📦 Dati raw conservati in:" -ForegroundColor Cyan
Write-Host "   - data/raw/"
Write-Host "   - data/raw_sample/"
Write-Host ""
Write-Host "🚀 Il progetto è pronto per una fresh run:" -ForegroundColor Cyan
Write-Host "   1. python -m src.data.make_dataset"
Write-Host "   2. python -m src.features.make_features"
Write-Host "   3. python -m src.training.train_model"
Write-Host "   4. python -m src.analytics.policy_rules"
Write-Host "   5. python -m src.analytics.client_profiling"
Write-Host "   6. python -m src.analytics.clustering"
Write-Host "   7. streamlit run app/streamlit_app.py"
Write-Host ""
