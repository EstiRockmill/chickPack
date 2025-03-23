using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace EasyPack.Data.Migrations
{
    /// <inheritdoc />
    public partial class MakeBoxIdNullable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Items_Boxes_BoxId",
                table: "Items");

            migrationBuilder.AlterColumn<int>(
                name: "BoxId",
                table: "Items",
                type: "int",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "int");

            migrationBuilder.AddForeignKey(
                name: "FK_Items_Boxes_BoxId",
                table: "Items",
                column: "BoxId",
                principalTable: "Boxes",
                principalColumn: "Id");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Items_Boxes_BoxId",
                table: "Items");

            migrationBuilder.AlterColumn<int>(
                name: "BoxId",
                table: "Items",
                type: "int",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "int",
                oldNullable: true);

            migrationBuilder.AddForeignKey(
                name: "FK_Items_Boxes_BoxId",
                table: "Items",
                column: "BoxId",
                principalTable: "Boxes",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }
    }
}
