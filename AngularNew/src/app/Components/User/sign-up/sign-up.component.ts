import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../../Service/auth.service';
import { SignUp } from '../../../Models/SignUp.model';

@Component({
  selector: 'app-sign-up',
  standalone: true,
  imports: [CommonModule, RouterLink, ReactiveFormsModule],
  templateUrl: './sign-up.component.html',
  styleUrl: './sign-up.component.css'
})
export class SignUpComponent implements OnInit {
  signUp!: FormGroup;
  loading = false;
  error = '';

  constructor(
    private authService: AuthService,
    private router: Router
  ) { }
  ngOnInit(): void {
    this.signUp = new FormGroup({
      name: new FormControl('', [
        Validators.required,
        Validators.minLength(2),
        Validators.maxLength(50),
        Validators.pattern(/^[a-zA-Zא-ת\s]+$/) 
      ]),
      email: new FormControl('', [
        Validators.required,
        Validators.email,
        Validators.pattern(/^[a-zA-Z0-9._%+-]+@gmail\.com$/)  
      ]),
      password: new FormControl('', [
        Validators.required,
        Validators.minLength(8),
        Validators.pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{8,}$/) 
      ]),
      address: new FormControl('', [
        Validators.required,
        Validators.minLength(5),
        Validators.maxLength(100),
        Validators.pattern(/^[a-zA-Zא-ת0-9\s,.'-]+$/) 
      ])
    });
  }

  onSubmit(): void {
    this.loading = true;
    this.error = '';

    this.authService.signUp(this.signUp.value)
      .subscribe({
        next: () => {
          this.router.navigate(['/room-selection']);
        },
        error: error => {
          this.error = error.message || 'Registration failed. Please try again.';
          this.loading = false;
        }
      });
  }
}
