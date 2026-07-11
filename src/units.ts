export type UnitSystem = "metric" | "imperial";

const cToF = (c: number) => c * 9 / 5 + 32;
const cDeltaToF = (c: number) => c * 9 / 5;

export function formatTempAnomaly(c: number, units: UnitSystem, digits = 1): string {
  if (units === "imperial") {
    const f = cDeltaToF(c);
    return `${f > 0 ? "+" : ""}${f.toFixed(digits)} F`;
  }
  return `${c > 0 ? "+" : ""}${c.toFixed(digits)} C`;
}

export function formatPrecip(mm: number, units: UnitSystem): number {
  return units === "imperial" ? mm / 25.4 : mm;
}

export function formatSnow(cm: number, units: UnitSystem): number {
  return units === "imperial" ? cm / 2.54 : cm;
}

export function precipLabel(units: UnitSystem): string {
  return units === "imperial" ? "in" : "mm";
}

export function snowLabel(units: UnitSystem): string {
  return units === "imperial" ? "in" : "cm";
}

export function tempLabel(units: UnitSystem): string {
  return units === "imperial" ? "F" : "C";
}
