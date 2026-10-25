// Shared by the real and mock APIs. STEPS matches STAGES in pet_recognition_system/app/server.py.
export const STEPS = ['detect', 'keypoints', 'behaviour', 'interaction'];
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

// Quick checks before sending; the server validates again.
export function checkUpload(file, recordedAt) {
  if (!file || !file.type.startsWith('video/')) throw new Error('Choose a video file');
  if (file.size > MAX_UPLOAD_BYTES) throw new Error('Video is larger than 500 MB');
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(recordedAt)) throw new Error('Set the recording date and time');
}
