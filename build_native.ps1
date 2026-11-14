$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$oldRustup=$env:RUSTUP_HOME
$oldCargo=$env:CARGO_HOME
$oldPath=$env:PATH
try {
    $env:RUSTUP_HOME=Join-Path $repoRoot '.tools/rustup'
    $env:CARGO_HOME=Join-Path $repoRoot '.tools/cargo'
    $env:PATH="$env:CARGO_HOME/bin;$env:PATH"
    $cargo=Join-Path $env:CARGO_HOME 'bin/cargo.exe'
    $manifest=Join-Path $PSScriptRoot 'native/Cargo.toml'
    & $cargo test --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw 'RustLander tests failed' }
    & $cargo build --release --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw 'RustLander build failed' }
} finally {
    $env:RUSTUP_HOME=$oldRustup
    $env:CARGO_HOME=$oldCargo
    $env:PATH=$oldPath
}
