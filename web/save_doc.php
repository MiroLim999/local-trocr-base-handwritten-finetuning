<?php
/**
 * save_doc.php
 * Saves a verified document as a PNG into ./saved_docs/.
 *
 * Expects JSON body:
 *   {
 *     "title": "Birth Certificate - Juan Dela Cruz",
 *     "docType": "birth",
 *     "image": "data:image/png;base64,...."
 *   }
 *
 * Returns: { "ok": true, "file": "<filename>.png", "meta": "<filename>.json" }
 */

header('Content-Type: application/json');

$saveDir = __DIR__ . DIRECTORY_SEPARATOR . 'saved_docs';
if (!is_dir($saveDir)) {
    mkdir($saveDir, 0777, true);
}

$raw = file_get_contents('php://input');
$data = json_decode($raw, true);

if (!$data || empty($data['image'])) {
    http_response_code(400);
    echo json_encode(['ok' => false, 'error' => 'Missing image data.']);
    exit;
}

$title   = isset($data['title']) ? trim($data['title']) : 'Document';
$docType = isset($data['docType']) ? preg_replace('/[^a-z]/', '', strtolower($data['docType'])) : 'document';
$image   = $data['image'];

// Strip the data URL prefix and decode.
if (strpos($image, ',') !== false) {
    $image = explode(',', $image, 2)[1];
}
$binary = base64_decode($image);
if ($binary === false) {
    http_response_code(400);
    echo json_encode(['ok' => false, 'error' => 'Invalid base64 image.']);
    exit;
}

// Build a safe, unique filename.
$safeTitle = preg_replace('/[^A-Za-z0-9._-]+/', '_', $title);
$safeTitle = trim($safeTitle, '_');
if ($safeTitle === '') {
    $safeTitle = 'document';
}
$stamp = date('Ymd_His');
$base  = $docType . '_' . $safeTitle . '_' . $stamp;
$pngName  = $base . '.png';
$jsonName = $base . '.json';

file_put_contents($saveDir . DIRECTORY_SEPARATOR . $pngName, $binary);

$meta = [
    'title'   => $title,
    'docType' => $docType,
    'file'    => $pngName,
    'savedAt' => date('c'),
    'fields'  => isset($data['fields']) ? $data['fields'] : [],
    'metrics' => isset($data['metrics']) ? $data['metrics'] : null,
];
file_put_contents(
    $saveDir . DIRECTORY_SEPARATOR . $jsonName,
    json_encode($meta, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE)
);

echo json_encode(['ok' => true, 'file' => $pngName, 'meta' => $jsonName]);
