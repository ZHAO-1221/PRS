// Class names match pet_recognition DOG_BEHAVIOR_CLASSES / INTERACTION_CLASSES.
export const DOG_BEHAVIOURS = {
  walking_run: { name: 'Walking / running', short: 'Walk/run', color: '#ffd45c' },
  standing: { name: 'Standing', short: 'Stand', color: '#6dc8c6' },
  sitting: { name: 'Sitting', short: 'Sit', color: '#8fa8e0' },
  lying_sleeping: { name: 'Lying / sleeping', short: 'Lie/sleep', color: '#9cc59a' },
  eating_drinking: { name: 'Eating / drinking', short: 'Eat/drink', color: '#d9503c' }
};

export const INTERACTIONS = {
  no_interaction: { name: 'No interaction', chip: '#ece7da' },
  hand_feeding: { name: 'Feeding', chip: '#ffd45c' },
  petting: { name: 'Petting', chip: '#b9e3e2' },
  playing_together: { name: 'Playing', chip: '#f6c4ba' },
  // pipeline label when the dog and the owner are not both in view
  insufficient_detection: { name: 'Alone', chip: '#ece7da' }
};

export const TOGETHER = ['hand_feeding', 'petting', 'playing_together'];
export const REST = ['lying_sleeping', 'sitting'];
export const ACTIVE = ['walking_run', 'standing', 'eating_drinking'];
