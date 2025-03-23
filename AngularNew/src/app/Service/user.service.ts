import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { User } from '../Models/User.model';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class UserService {
  constructor(private _httpClient: HttpClient) { }
  getFromServer(): Observable<User[]> {
    return this._httpClient.get<User[]>(
      `https://localhost:7223/api/User`);
  }
  getByIdFromServer(id: number): Observable<User> {
    return this._httpClient.get<User>(
      `https://localhost:7223/api/User${id}`
    );
  }

  putServer(id: number, i: User): Observable<any> {
    return this._httpClient.put<any>(`https://localhost:7223/api/User${id}`, i);
  }

  deleteServer(id: number): Observable<any> {
    return this._httpClient.delete<any>(`https://localhost:7223/api/User${id}`);
  }

  addServer(u: User): Observable<any> {
    return this._httpClient.post<any>(`https://localhost:7223/api/User`, u);
  }
}
