<?php
/**
 * delete_doc.php
 * Deletes a saved document (its PNG and matching .json metadata).
 *
 * Expects JSON body: { "file": "birth_xxx_20260101_120000.png" }
 * Returns: { "ok": true }
 */

header('Content-Type: application/json');

$saveDir = __DIR__ . DIRECTORY_SEPARATOR . 'saved_docs';

$raw = file_get_contents('php://input');
$data = json_decode($raw, true);
$file = isset($data['file']) ? basename($data['file']) : '';

if ($file === '' || substr($file, -4) !== '.png') {
    http_response_code(400);
    echo json_encode(['ok' => false, 'error' => 'Invalid file.']);
    exit;
}

$png  = $saveDir . DIRECTORY_SEPARATOR . $file;
$json = $saveDir . DIRECTORY_SEPARATOR . substr($file, 0, -4) . '.json';

if (file_exists($png))  { @unlink($png); }
if (file_exists($json)) { @unlink($json); }

echo json_encode(['ok' => true]);
