// Runtime configuration for the frontend.
window.APP_CONFIG = {
  // Base URL of the Python TrOCR Flask API (see ../api/app.py).
  OCR_API: 'http://127.0.0.1:5000',

  // Field templates per document type. Positions are fractions (0-1) of the
  // page width/height, used to auto-place draggable boxes the user can adjust.
  FIELD_TEMPLATES: {
    birth: [
      { name: 'Child Full Name',  x: 0.30, y: 0.28, w: 0.45, h: 0.05 },
      { name: 'Date of Birth',    x: 0.30, y: 0.37, w: 0.30, h: 0.05 },
      { name: 'Sex',              x: 0.30, y: 0.46, w: 0.18, h: 0.05 },
      { name: 'Place of Birth',   x: 0.30, y: 0.55, w: 0.40, h: 0.05 },
      { name: 'Father Full Name', x: 0.30, y: 0.64, w: 0.45, h: 0.05 },
      { name: 'Mother Full Name', x: 0.30, y: 0.73, w: 0.45, h: 0.05 },
    ],
    death: [
      { name: 'Full Name',        x: 0.30, y: 0.28, w: 0.45, h: 0.05 },
      { name: 'Date of Death',    x: 0.30, y: 0.37, w: 0.30, h: 0.05 },
      { name: 'Sex',              x: 0.30, y: 0.46, w: 0.18, h: 0.05 },
      { name: 'Place of Death',   x: 0.30, y: 0.55, w: 0.40, h: 0.05 },
      { name: 'Cause of Death',   x: 0.30, y: 0.64, w: 0.45, h: 0.05 },
    ],
    marriage: [
      { name: 'Husband Full Name', x: 0.30, y: 0.28, w: 0.45, h: 0.05 },
      { name: 'Wife Full Name',    x: 0.30, y: 0.37, w: 0.45, h: 0.05 },
      { name: 'Date of Marriage',  x: 0.30, y: 0.46, w: 0.30, h: 0.05 },
      { name: 'Place of Marriage', x: 0.30, y: 0.55, w: 0.40, h: 0.05 },
    ],
  },

  DOC_LABELS: {
    birth: 'Birth Certificate',
    death: 'Death Certificate',
    marriage: 'Marriage Certificate',
  },
};
