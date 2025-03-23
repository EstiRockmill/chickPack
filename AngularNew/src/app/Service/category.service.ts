import { Injectable } from '@angular/core';
import { Category } from '../Models/Category.model';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CategoryService {
  
  constructor(private _httpClient:HttpClient) { }
getFromServer(): Observable<Category[]> {
    return this._httpClient.get<Category[]>(
    `https://localhost:7223/api/Category`    );
  }
getByIdFromServer(id: number): Observable<Category> {
    return this._httpClient.get<Category>(
      `https://localhost:7223/api/Category${id}`
    );
  }

  putServer(id: number, c: Category): Observable<any>{
    return this._httpClient.put<any>(`https://localhost:7223/api/Category${id}`, c);
  }

  deleteServer(id: number) :Observable<any>{
    return this._httpClient.delete<any>(`https://localhost:7223/api/Category${id}`);
  }

  addServer(c: Category) :Observable<any>{
   return this._httpClient.post<any>(`https://localhost:7223/api/Category`, c);
  }}
