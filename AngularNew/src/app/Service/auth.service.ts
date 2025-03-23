import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Login } from '../Models/Login.model';
import { BehaviorSubject, Observable } from 'rxjs';
import { SignUp } from '../Models/SignUp.model';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  
  isAuthenticated$ = new BehaviorSubject<boolean>(false); 

  constructor(private _httpClient: HttpClient) {
    if (typeof window !== 'undefined') {
      const isLoggedIn = !!localStorage.getItem('appSession');
      this.isAuthenticated$.next(isLoggedIn);
    }
  }
  get isAuthenticated(): boolean {
    return this.isAuthenticated$.getValue();
  }
  getCurrentUser(): any {
    const session = localStorage.getItem('appSession');
    return session ? JSON.parse(session).user : null;
  }
  getCurrentUserId(): number  {
    const user = this.getCurrentUser();
    return user.id;
  }

  login(l: Login): Observable<any> {
    return new Observable(observer => {
      this._httpClient.post<any>(`https://localhost:7223/api/Auth/login`, l).subscribe({
        next: (res) => {
          if (res && res.token) {
            localStorage.setItem('appSession', JSON.stringify({ user: res.user, token: res.token }));
            this.isAuthenticated$.next(true);
          }
          observer.next(res.status);
          observer.complete();
        },
        error: (err) => {
          console.error("Login failed", err);
          observer.error(err);
        }
      });
    });
  }
  signUp(s: SignUp): Observable<any> {
    return new Observable(observer => {
      this._httpClient.post<any>(`https://localhost:7223/api/Auth/signup`, s).subscribe({
        next: (res) => {
          if (res && res.token) {
            localStorage.setItem('appSession', JSON.stringify({ user: res.user, token: res.token }));
            this.isAuthenticated$.next(true);
            console.log("User signed up successfully!", res);
          }
          observer.next(res.status);
          observer.complete();
        },
        error: (err) => {
          console.error("Sign up failed", err);
          observer.error(err);
        }
      });
    });
  }
  

  logout(): void {
    localStorage.removeItem('appSession');
    this.isAuthenticated$.next(false);
  }
  getToken(): string | null {
    const session = localStorage.getItem('appSession');
    return session ? JSON.parse(session).token : null;
  }
  

 
}
