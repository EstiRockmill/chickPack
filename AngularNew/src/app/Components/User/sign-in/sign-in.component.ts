import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../../Service/auth.service';

@Component({
  selector: 'app-sign-in',
  standalone: true,
  imports: [CommonModule, RouterLink, ReactiveFormsModule],
  templateUrl: './sign-in.component.html',
  styleUrl: './sign-in.component.css'
})
export class SignInComponent implements OnInit {
  login!: FormGroup;
  loading = false;
  error = '';

  constructor(
    private authService: AuthService,
    private router: Router
  ) { }
  ngOnInit(): void {
    this.login = new FormGroup({
      email: new FormControl('', [
        Validators.required,
        Validators.email,
        Validators.pattern(/^[a-zA-Z0-9._%+-]+@gmail\.com$/)
      ]),
      password: new FormControl('', [
        Validators.required,
        Validators.minLength(8),
        Validators.pattern(/^(?=.*[a-zA-Z])(?=.*[!@#$%^&*(),.?":{}|<>])[0-9a-zA-Z!@#$%^&*(),.?":{}|<>]{8,}$/)
      ])
    })
  }

  onSubmit(): void {
    this.loading = true;
    this.error = '';

    this.authService.login(this.login.value)
      .subscribe({
        next: () => {
          this.router.navigate(['/room-selection']);
        },
        error: error => {
          if (error.status === 404) {
            this.router.navigate(['/signUp'])
          } else if (error.status === 401) {
            this.error = 'Incorrect password. Please try again.';
          } else {
            this.error = 'An unexpected error occurred. Please try again later.';
          }
          this.loading = false;

        }
      });
  }
}
