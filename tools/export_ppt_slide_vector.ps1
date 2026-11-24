param(
    [Parameter(Mandatory = $true)][string]$InputPptx,
    [Parameter(Mandatory = $true)][int]$SlideIndex,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [double]$PaddingPoints = 24
)

$ErrorActionPreference = 'Stop'
$powerPoint = $null
$sourcePresentation = $null
$targetPresentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $sourcePresentation = $powerPoint.Presentations.Open($InputPptx, $true, $false, $false)
    if ($SlideIndex -lt 1 -or $SlideIndex -gt $sourcePresentation.Slides.Count) {
        throw "Slide index $SlideIndex is outside 1..$($sourcePresentation.Slides.Count)"
    }
    $sourceSlide = $sourcePresentation.Slides.Item($SlideIndex)

    # PowerPoint connectors can report bounds far larger than their visible line.
    # The non-connector objects define the intended figure canvas.
    $left = [double]::PositiveInfinity
    $top = [double]::PositiveInfinity
    $right = [double]::NegativeInfinity
    $bottom = [double]::NegativeInfinity
    foreach ($shape in $sourceSlide.Shapes) {
        if ($shape.Visible -eq -1 -and $shape.Connector -ne -1) {
            $left = [Math]::Min($left, [double]$shape.Left)
            $top = [Math]::Min($top, [double]$shape.Top)
            $right = [Math]::Max($right, [double]$shape.Left + [double]$shape.Width)
            $bottom = [Math]::Max($bottom, [double]$shape.Top + [double]$shape.Height)
        }
    }
    if ([double]::IsInfinity($left)) {
        throw "Slide $SlideIndex has no visible non-connector shapes"
    }

    # Size a fresh presentation before pasting so PowerPoint cannot reflow the
    # original slide while its page dimensions are changing.
    $targetPresentation = $powerPoint.Presentations.Add($false)
    $targetPresentation.PageSetup.SlideWidth = ($right - $left) + (2 * $PaddingPoints)
    $targetPresentation.PageSetup.SlideHeight = ($bottom - $top) + (2 * $PaddingPoints)
    $targetSlide = $targetPresentation.Slides.Add(1, 12)
    $clonedDesign = $targetPresentation.Designs.Clone($sourceSlide.Design)
    $targetSlide.Design = $clonedDesign
    $sourceSlide.Shapes.Range().Copy()
    [void]$targetSlide.Shapes.Paste()
    $deltaX = $PaddingPoints - $left
    $deltaY = $PaddingPoints - $top
    foreach ($shape in $targetSlide.Shapes) {
        $shape.Left = [double]$shape.Left + $deltaX
        $shape.Top = [double]$shape.Top + $deltaY
    }

    $outputDirectory = Split-Path -Parent $OutputPdf
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
    $targetPresentation.SaveAs($OutputPdf, 32)
}
finally {
    if ($targetPresentation -ne $null) {
        $targetPresentation.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($targetPresentation)
    }
    if ($sourcePresentation -ne $null) {
        $sourcePresentation.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($sourcePresentation)
    }
    if ($powerPoint -ne $null) {
        $powerPoint.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
