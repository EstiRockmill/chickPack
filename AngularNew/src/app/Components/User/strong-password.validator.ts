import { AbstractControl, ValidationErrors } from '@angular/forms';

export function strongPasswordValidator(control: AbstractControl): ValidationErrors | null {
  const password = control.value;

  if (!password) {
    return null; 
  }

  const hasLetter = /[a-zA-Z]/.test(password);
  const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  const hasOnlyNumbersAndOneLetter = /^[0-9]*[a-zA-Z][0-9]*$/.test(password);
  const isLongEnough = password.length >= 8;

  if (!hasLetter || !hasSpecialChar || !hasOnlyNumbersAndOneLetter || !isLongEnough) {
    return { strongPasswordValidator: true };
  }

  return null; 
}
