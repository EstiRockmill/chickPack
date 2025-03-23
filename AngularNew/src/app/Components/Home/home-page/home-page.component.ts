import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../../Service/auth.service';

@Component({
  selector: 'app-home-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.css'
})
export class HomePageComponent {
  isAuthenticated = false;
  
  constructor(private authService: AuthService,private router:Router) {
    this.isAuthenticated = this.authService.isAuthenticated;
  }
  getStarted() {
    if (this.authService.isAuthenticated) {
      this.router.navigate(['/room-selection']);
    } else {
      this.router.navigate(['/login']);
    }
  }
}
