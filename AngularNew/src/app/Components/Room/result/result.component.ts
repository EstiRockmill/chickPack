import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, ViewChild, ElementRef } from '@angular/core';
import { OptimizeRequest } from '../../../Models/OptimizeRequest.model';
import { OptimizeResultService } from '../../../Service/optimize-result.service';
import { Room } from '../../../Models/Room.model';
import { ImageAnalsys } from '../../../Models/ImageAnalsys.model';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { Router } from '@angular/router';

@Component({
  selector: 'app-result',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './result.component.html',
  styleUrl: './result.component.css'
})
export class ResultComponent {
  @Input() analysisResults: ImageAnalsys[] = [];
  @Input() selectedBoxVolume: number = 0;
  @Input() room!: Room;
  @Output() back = new EventEmitter<void>();

  error!: Error;
  loading: boolean = false;
  packingResult!: Room;
  openBoxes: boolean[] = [];

  @ViewChild('pdfContent', { static: false }) pdfContent!: ElementRef;

  constructor(private optimizeResultService: OptimizeResultService,private router:Router) {}

  ngOnInit(): void {
    this.OptimizeCalculation();
  }

  OptimizeCalculation() {
    const requestData: OptimizeRequest = {
      capacity: this.selectedBoxVolume,
      roomId: this.room.id,
      itemJsons: this.analysisResults
    };

    this.loading = true;
    this.optimizeResultService.OptimizeCalculation(requestData).subscribe({
      next: (result: Room) => {
        if ((result as any).boxes?.$values) {
          result.boxes = (result as any).boxes.$values;
          result.boxes.forEach((box: any) => {
            if (box.items?.$values) {
              box.items = box.items.$values;
            }
          });
        }

        this.packingResult = result;
        this.openBoxes = new Array(this.packingResult.boxes.length).fill(false);
        this.loading = false;
      },
      error: (err) => {
        this.error = err;
        this.loading = false;
      }
    });
  }

  toggleBox(index: number) {
    this.openBoxes[index] = !this.openBoxes[index];
  }

  goBack() {
    this.back.emit();
  }


  exportToPDF(): void {
    const doc = new jsPDF();
    const title = `Optimized Packing Plan for ${this.packingResult.name}`;
    doc.setFontSize(16);
    doc.text(title, 14, 20);
    doc.setFontSize(12);
    doc.text(`Total Boxes Required: ${this.packingResult.numOfBoxes}`, 14, 30);
  
    let currentY = 40;
  
    this.packingResult.boxes.forEach((box, index) => {
      const boxTitle = `Box ${index + 1}`;
      const items = box.items.map(item => [item.name]);
  
      autoTable(doc, {
        head: [[boxTitle]],
        body: items,
        startY: currentY,
        theme: 'grid',
        styles: {
          fontSize: 11,
          halign: 'left'
        },
        headStyles: {
          fillColor: [63, 81, 181],
          textColor: 255,
          fontStyle: 'bold'
        },
        margin: { left: 14, right: 14 },
        didDrawPage: (data) => {
          currentY = (data.cursor?.y ?? currentY) + 10;
        }
      });
    });
  
    doc.save('OptimizedPackingPlan.pdf');
  }
  goBackToRoomSelection() {
    this.router.navigate(['/room-selection']); 
  }
  
  
}
