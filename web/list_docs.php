<?php
/**
 * list_docs.php
 * Returns the list of saved documents (newest first) from ./saved_docs/.
 *
 * Returns: { "ok": true, "docs": [ { title, docType, file, url, savedAt, metrics } ] }
 */

header('Content-Type: application/json');

$saveDir = __DIR__ . DIRECTORY_SEPARATOR . 'saved_docs';
$docs = [];

if (is_dir($saveDir)) {
    $metaFiles = glob($saveDir . DIRECTORY_SEPARATOR . '*.json');
    foreach ($metaFiles as $metaPath) {
        $meta = json_decode(file_get_contents($metaPath), true);
        if (!$meta || empty($meta['file'])) {
            continue;
        }
        $pngPath = $saveDir . DIRECTORY_SEPARATOR . $meta['file'];
        if (!file_exists($pngPath)) {
            continue;
        }
        $docs[] = [
            'title'   => isset($meta['title']) ? $meta['title'] : $meta['file'],
            'docType' => isset($meta['docType']) ? $meta['docType'] : 'document',
            'file'    => $meta['file'],
            'url'     => 'saved_docs/' . rawurlencode($meta['file']),
            'savedAt' => isset($meta['savedAt']) ? $meta['savedAt'] : null,
            'metrics' => isset($meta['metrics']) ? $meta['metrics'] : null,
            '_mtime'  => filemtime($metaPath),
        ];
    }
    // Newest first.
    usort($docs, function ($a, $b) {
        return $b['_mtime'] <=> $a['_mtime'];
    });
    foreach ($docs as &$d) {
        unset($d['_mtime']);
    }
}

echo json_encode(['ok' => true, 'docs' => $docs]);
