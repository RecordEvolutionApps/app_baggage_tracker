export function getRandomColor() {
  // Generate random values for red, green, and blue channels
  const red = Math.floor(Math.random() * 256);
  const green = Math.floor(Math.random() * 256);
  const blue = Math.floor(Math.random() * 256);

  // Construct the color string in hexadecimal format
  const color =
    '#' +
    red.toString(16).padStart(2, '0') +
    green.toString(16).padStart(2, '0') +
    blue.toString(16).padStart(2, '0');

  return color;
}

export function hexToTransparent(hex: string, opacity: number) {
  // Remove the hash if it exists
  hex = hex.replace('#', '');

  // Parse the hex value into RGB components
  var r = parseInt(hex.substring(0, 2), 16);
  var g = parseInt(hex.substring(2, 4), 16);
  var b = parseInt(hex.substring(4, 6), 16);

  // Convert opacity to a value between 0 and 1
  opacity = Math.min(Math.max(opacity, 0), 1);

  // Convert the RGB values into a transparent RGBA string
  var rgba = 'rgba(' + r + ', ' + g + ', ' + b + ', ' + opacity + ')';

  return rgba;
}
