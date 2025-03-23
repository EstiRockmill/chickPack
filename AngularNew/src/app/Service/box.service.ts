import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Box } from '../Models/Box.model';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class BoxService {

  constructor(private _httpClient:HttpClient) { }
  getFromServer(): Observable<Box[]> {
    return this._httpClient.get<Box[]>(
    `https://localhost:7223/api/Box`    );
  }
  getByIdFromServer(id: number): Observable<Box> {
    return this._httpClient.get<Box>(
      `https://localhost:7223/api/Box${id}`
    );
  }

  putServer(id: number, b: Box): Observable<any>{
    return this._httpClient.put<any>(`https://localhost:7223/api/Box${id}`, b);
  }

  deleteServer(id: number) :Observable<any>{
    return this._httpClient.delete<any>(`https://localhost:7223/api/Box${id}`);
  }

  addServer(b: Box) :Observable<any>{
   return this._httpClient.post<any>(`https://localhost:7223/api/Box`, b);
  }
}
