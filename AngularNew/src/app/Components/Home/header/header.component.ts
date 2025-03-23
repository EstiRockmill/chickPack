import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { User } from '../../../Models/User.model';
import { AuthService } from '../../../Service/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {
  currentUser: User | null = null;
  constructor(private authService: AuthService,private router:Router) {}
  
  ngOnInit(): void {
    // מנוי למעקב אחרי שינויים במצב התחברות
    this.authService.isAuthenticated$.subscribe(isAuth => {
      if (isAuth) {
        this.currentUser = this.authService.getCurrentUser();
      } else {
        this.currentUser = null;
      }
    });
    if (this.authService.isAuthenticated) {
      this.currentUser = this.authService.getCurrentUser();
    }
  }
  logout(): void {
    this.authService.logout();
    this.currentUser = null; 
    this.router.navigate(['/']);
  }
}
